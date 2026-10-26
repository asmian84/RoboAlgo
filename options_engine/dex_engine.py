"""
options_engine/dex_engine.py
────────────────────────────────────────────────────────────────────────────────
Delta Exposure (DEX) Engine.

DEX measures net dealer DELTA exposure — how much the aggregate market maker
book needs to hedge in underlying shares as price moves.

Formula (dealer perspective):
  call_dex = +delta × call_OI × CONTRACT_SIZE    (dealer SELLS calls → long delta)
  put_dex  = -delta × put_OI  × CONTRACT_SIZE    (dealer SELLS puts  → short delta)
  net_dex  = call_dex + put_dex                  (net share-equivalent exposure)

Positive DEX → dealers net long delta → sell on rallies (bearish pressure)
Negative DEX → dealers net short delta → buy on dips (bullish support)

Output:
  {
    "symbol":     str,
    "spot":       float,
    "total_dex":  float,          # net share-equivalent exposure
    "dex_by_strike": {            # per-strike breakdown
        strike: { call_dex, put_dex, net_dex }
    },
    "dex_regime": str,            # "LONG_DELTA" | "SHORT_DELTA" | "NEUTRAL"
    "dealer_bias": int,           # +1 bullish, -1 bearish, 0 neutral
    "zero_delta_level": float,    # strike where net dealer delta = 0
    "dex_score": float,           # 0-100 normalized score for scanner
  }
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from data_engine.yahoo_options_fetcher import OptionsChain

logger = logging.getLogger(__name__)

CONTRACT_SIZE = 100


@dataclass
class DEXProfile:
    symbol:           str
    spot:             float
    total_dex:        float        # net dealer delta (share-equivalents)
    total_call_dex:   float
    total_put_dex:    float
    dex_by_strike:    list[dict] = field(default_factory=list)
    dex_regime:       str = "NEUTRAL"    # LONG_DELTA | SHORT_DELTA | NEUTRAL
    dealer_bias:      int = 0            # +1 | -1 | 0
    zero_delta_level: float = 0.0
    dex_score:        float = 50.0       # 0-100

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol":           self.symbol,
            "spot":             self.spot,
            "total_dex":        round(self.total_dex, 0),
            "total_call_dex":   round(self.total_call_dex, 0),
            "total_put_dex":    round(self.total_put_dex, 0),
            "dex_by_strike":    self.dex_by_strike,
            "dex_regime":       self.dex_regime,
            "dealer_bias":      self.dealer_bias,
            "zero_delta_level": round(self.zero_delta_level, 2),
            "dex_score":        round(self.dex_score, 1),
        }


class DEXEngine:
    """Compute Delta Exposure (DEX) from an OptionsChain."""

    def __init__(self, near_expiry_only: bool = False, max_expiry_days: int = 60):
        self.near_expiry_only = near_expiry_only
        self.max_expiry_days  = max_expiry_days

    def compute(self, chain: "OptionsChain") -> DEXProfile:
        """Compute full DEX profile from an OptionsChain."""
        from datetime import datetime, timezone

        spot    = chain.spot_price
        today   = datetime.now(timezone.utc)
        strikes: dict[float, dict] = {}

        for contract in chain.contracts:
            # Expiry filter
            if self.near_expiry_only:
                try:
                    exp = datetime.fromisoformat(contract.expiration).replace(
                        tzinfo=timezone.utc
                    )
                    days_to_exp = (exp - today).days
                    if days_to_exp > self.max_expiry_days:
                        continue
                except Exception:
                    pass

            strike = contract.strike
            delta  = contract.delta
            oi     = contract.open_interest

            if delta == 0 or oi == 0:
                continue

            if strike not in strikes:
                strikes[strike] = {
                    "strike":    strike,
                    "call_dex":  0.0,
                    "put_dex":   0.0,
                    "net_dex":   0.0,
                }

            if contract.option_type == "call":
                # Dealer sold call → long delta position → +delta × OI × 100
                dex = float(delta) * oi * CONTRACT_SIZE
                strikes[strike]["call_dex"] += dex
            else:
                # Dealer sold put → short delta position → -delta × OI × 100
                # Put delta is negative, so -(negative delta) = positive for OTM puts
                # But dealer SOLD put → dealer is SHORT delta = short (for ITM puts)
                dex = -abs(float(delta)) * oi * CONTRACT_SIZE
                strikes[strike]["put_dex"] += dex

        if not strikes:
            return DEXProfile(
                symbol=chain.symbol, spot=spot,
                total_dex=0, total_call_dex=0, total_put_dex=0,
            )

        # Net per strike
        strike_list = []
        for s, row in sorted(strikes.items()):
            row["net_dex"] = row["call_dex"] + row["put_dex"]
            strike_list.append({
                "strike":   s,
                "call_dex": round(row["call_dex"], 0),
                "put_dex":  round(row["put_dex"], 0),
                "net_dex":  round(row["net_dex"], 0),
            })

        total_call = sum(r["call_dex"] for r in strike_list)
        total_put  = sum(r["put_dex"]  for r in strike_list)
        total_net  = total_call + total_put

        # Zero-delta level: strike where cumulative DEX crosses zero
        zero_level = self._find_zero_delta(strike_list, spot)

        # Regime
        if total_net > 0:
            regime = "LONG_DELTA"     # dealers net long → sell pressure on up moves
            bias   = -1               # bearish (dealers hedge by selling)
        elif total_net < 0:
            regime = "SHORT_DELTA"    # dealers net short → buy pressure on down moves
            bias   = 1                # bullish (dealers hedge by buying)
        else:
            regime = "NEUTRAL"
            bias   = 0

        # Score 0-100: magnitude of DEX relative to typical range
        abs_dex = abs(total_net)
        # Normalize against rough market scale: 1M shares = 50 score
        dex_score = float(np.clip(abs_dex / 20_000 * 50, 0, 100))

        return DEXProfile(
            symbol           = chain.symbol,
            spot             = spot,
            total_dex        = round(total_net, 0),
            total_call_dex   = round(total_call, 0),
            total_put_dex    = round(total_put, 0),
            dex_by_strike    = strike_list,
            dex_regime       = regime,
            dealer_bias      = bias,
            zero_delta_level = zero_level,
            dex_score        = round(dex_score, 1),
        )

    @staticmethod
    def _find_zero_delta(strike_list: list[dict], spot: float) -> float:
        """Find strike where cumulative net DEX crosses zero."""
        if not strike_list:
            return spot
        cum = 0.0
        prev_strike = strike_list[0]["strike"]
        for row in strike_list:
            cum += row["net_dex"]
            if cum >= 0:
                return float((prev_strike + row["strike"]) / 2)
            prev_strike = row["strike"]
        return float(strike_list[-1]["strike"])


# ── Convenience function ──────────────────────────────────────────────────────

def compute_dex(symbol: str, near_expiry_only: bool = False) -> dict[str, Any]:
    """
    Full pipeline: fetch options chain → compute DEX.
    Returns dict ready for API/JSON.
    """
    from data_engine.yahoo_options_fetcher import OptionsChainFetcher
    fetcher = OptionsChainFetcher()
    chain   = fetcher.fetch(symbol)
    if chain is None:
        return {"symbol": symbol, "error": "options data unavailable"}
    engine  = DEXEngine(near_expiry_only=near_expiry_only)
    profile = engine.compute(chain)
    return profile.to_dict()


# ── Example ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    result = compute_dex("SPY")
    print(json.dumps({k: v for k, v in result.items() if k != "dex_by_strike"}, indent=2))
    print(f"\nTop DEX strikes:")
    for row in sorted(result.get("dex_by_strike", []),
                      key=lambda x: abs(x["net_dex"]), reverse=True)[:5]:
        print(f"  ${row['strike']}: net={row['net_dex']:,.0f}  call={row['call_dex']:,.0f}  put={row['put_dex']:,.0f}")
