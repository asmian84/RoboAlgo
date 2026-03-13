"""
RoboAlgo — Execution Simulator
Models realistic order execution: slippage, spread, partial fills.

Execution model:
  Market order slippage  = price × SLIPPAGE_PCT (per side, entry and exit)
  Commission             = shares × COMMISSION_PER_SHARE
  Spread cost            = price × SPREAD_PCT / 2 (ask-side for buys, bid-side for sells)
  Partial fill           = randomly fills 70–100% of requested shares on illiquid instruments

Liquidity tiers (by avg daily volume):
  HIGH    (>= 5M shares/day)   → full fill, minimal slippage
  MEDIUM  (1M–5M shares/day)   → 95% fill rate, 0.08% slippage
  LOW     (< 1M shares/day)    → 80–95% fill rate, 0.15–0.25% slippage
"""

import logging
import random
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Execution Constants ────────────────────────────────────────────────────────
SLIPPAGE_PCT         = 0.001   # 0.10% per side (base)
SPREAD_PCT           = 0.001   # 0.10% spread (full)
COMMISSION_PER_SHARE = 0.005   # $0.005 / share

# Liquidity tier thresholds (avg daily volume in shares)
LIQUIDITY_HIGH_THRESHOLD   = 5_000_000
LIQUIDITY_MEDIUM_THRESHOLD = 1_000_000

# Partial fill parameters by liquidity tier
FILL_RATES = {
    "HIGH":   (1.00, 1.00),    # always full
    "MEDIUM": (0.95, 1.00),    # 95–100%
    "LOW":    (0.80, 0.95),    # 80–95%
}
SLIPPAGE_MULT = {
    "HIGH":   1.0,
    "MEDIUM": 0.8,
    "LOW":    1.5,
}


@dataclass
class ExecutionResult:
    """Complete result of simulating an order execution."""
    symbol:           str
    side:             str          # "BUY" or "SELL"
    requested_shares: int
    filled_shares:    int
    fill_pct:         float        # fraction filled (0–1)

    requested_price:  float        # mid-price at time of order
    executed_price:   float        # price after slippage + spread
    slippage_cost:    float        # total slippage dollars
    spread_cost:      float        # total spread cost dollars
    commission:       float        # total commission dollars
    total_cost:       float        # all-in execution cost (slippage + spread + commission)

    liquidity_tier:   str          # HIGH / MEDIUM / LOW
    partial_fill:     bool
    notes:            list[str]


class ExecutionSimulator:
    """
    Simulates realistic order execution with slippage, spread, and partial fills.

    Usage:
        sim = ExecutionSimulator()
        result = sim.simulate_entry("SOXL", shares=100, price=62.50, avg_volume=8_000_000)
        result = sim.simulate_exit("SOXL", shares=100, price=65.00, avg_volume=8_000_000)
        cost_summary = sim.estimate_round_trip("SOXL", shares=100, entry=62.50, exit=65.00, avg_volume=8_000_000)
    """

    def simulate_entry(
        self,
        symbol: str,
        shares: int,
        price: float,
        avg_volume: float = 5_000_000,
        order_type: Literal["market", "limit"] = "market",
    ) -> ExecutionResult:
        """
        Simulate a BUY order execution.
        Market orders pay the ask (price + spread/2 + slippage).
        Limit orders get a better fill but risk non-execution.
        """
        return self._simulate_order(
            symbol=symbol,
            side="BUY",
            shares=shares,
            price=price,
            avg_volume=avg_volume,
            order_type=order_type,
        )

    def simulate_exit(
        self,
        symbol: str,
        shares: int,
        price: float,
        avg_volume: float = 5_000_000,
        order_type: Literal["market", "limit"] = "market",
    ) -> ExecutionResult:
        """
        Simulate a SELL order execution.
        Market orders hit the bid (price - spread/2 - slippage).
        """
        return self._simulate_order(
            symbol=symbol,
            side="SELL",
            shares=shares,
            price=price,
            avg_volume=avg_volume,
            order_type=order_type,
        )

    def estimate_round_trip(
        self,
        symbol: str,
        shares: int,
        entry_price: float,
        exit_price: float,
        avg_volume: float = 5_000_000,
    ) -> dict:
        """
        Estimate total round-trip execution cost (entry + exit).
        Returns gross P&L, net P&L, breakeven distance, and cost breakdown.
        """
        entry = self.simulate_entry(symbol, shares, entry_price, avg_volume)
        exit_ = self.simulate_exit(symbol, entry.filled_shares, exit_price, avg_volume)

        gross_pnl = (exit_price - entry_price) * entry.filled_shares
        total_cost = entry.total_cost + exit_.total_cost
        net_pnl    = gross_pnl - total_cost

        # Minimum move to break even on costs
        breakeven_move = total_cost / max(entry.filled_shares, 1)
        breakeven_pct  = breakeven_move / max(entry_price, 1e-8)

        return {
            "symbol":          symbol,
            "shares":          entry.filled_shares,
            "entry_price":     entry.executed_price,
            "exit_price":      exit_.executed_price,
            "gross_pnl":       round(gross_pnl, 4),
            "total_cost":      round(total_cost, 4),
            "net_pnl":         round(net_pnl, 4),
            "breakeven_move":  round(breakeven_move, 4),
            "breakeven_pct":   round(breakeven_pct, 6),
            "entry_result":    _result_to_dict(entry),
            "exit_result":     _result_to_dict(exit_),
        }

    def apply_to_signal(
        self,
        signal: dict,
        shares: int,
        avg_volume: float = 5_000_000,
    ) -> dict:
        """
        Apply execution simulation to a signal dict.
        Adjusts entry, tier1, tier2, tier3 prices for realistic execution costs.
        Returns the signal enriched with execution cost estimates.
        """
        symbol     = signal.get("symbol", "UNKNOWN")
        entry_px   = float(signal.get("entry_price", 0) or 0)
        tier1_px   = float(signal.get("tier1_sell", 0) or 0)
        tier2_px   = float(signal.get("tier2_sell", 0) or 0)
        tier3_px   = float(signal.get("tier3_hold", 0) or 0)

        tier1_shares = shares // 3
        tier2_shares = shares // 3
        tier3_shares = shares - tier1_shares - tier2_shares

        entry_ex  = self.simulate_entry(symbol, shares, entry_px, avg_volume)

        results = {
            "execution": {
                "entry":      _result_to_dict(entry_ex),
                "effective_entry_price": entry_ex.executed_price,
            },
            "cost_breakdown": {
                "entry_slippage":  round(entry_ex.slippage_cost, 4),
                "entry_spread":    round(entry_ex.spread_cost, 4),
                "entry_commission": round(entry_ex.commission, 4),
            },
        }

        # Estimate exit costs for each tier
        if tier1_px > 0 and tier1_shares > 0:
            t1 = self.simulate_exit(symbol, tier1_shares, tier1_px, avg_volume)
            results["execution"]["tier1"] = _result_to_dict(t1)
        if tier2_px > 0 and tier2_shares > 0:
            t2 = self.simulate_exit(symbol, tier2_shares, tier2_px, avg_volume)
            results["execution"]["tier2"] = _result_to_dict(t2)
        if tier3_px > 0 and tier3_shares > 0:
            t3 = self.simulate_exit(symbol, tier3_shares, tier3_px, avg_volume)
            results["execution"]["tier3"] = _result_to_dict(t3)

        return {**signal, **results}

    # ── Internal ───────────────────────────────────────────────────────────────

    def _simulate_order(
        self,
        symbol: str,
        side: str,
        shares: int,
        price: float,
        avg_volume: float,
        order_type: str,
    ) -> ExecutionResult:
        notes = []
        tier  = self._liquidity_tier(avg_volume)

        # ── Partial Fill ───────────────────────────────────────────────────────
        fill_min, fill_max = FILL_RATES[tier]
        fill_pct    = random.uniform(fill_min, fill_max)
        filled      = max(1, int(shares * fill_pct))
        partial     = filled < shares
        if partial:
            notes.append(f"Partial fill: {filled}/{shares} shares ({fill_pct:.1%}) — {tier} liquidity")

        # ── Slippage ───────────────────────────────────────────────────────────
        slip_pct    = SLIPPAGE_PCT * SLIPPAGE_MULT[tier]
        if order_type == "limit":
            slip_pct *= 0.25   # limit orders have much lower slippage
            notes.append("Limit order: reduced slippage")

        if side == "BUY":
            slipped_price = price * (1 + slip_pct + SPREAD_PCT / 2)
        else:
            slipped_price = price * (1 - slip_pct - SPREAD_PCT / 2)

        slippage_cost = abs(slipped_price - price) * filled - (SPREAD_PCT / 2 * price * filled)
        spread_cost   = SPREAD_PCT / 2 * price * filled
        commission    = filled * COMMISSION_PER_SHARE
        total_cost    = slippage_cost + spread_cost + commission

        return ExecutionResult(
            symbol=symbol,
            side=side,
            requested_shares=shares,
            filled_shares=filled,
            fill_pct=fill_pct,
            requested_price=round(price, 4),
            executed_price=round(slipped_price, 4),
            slippage_cost=round(slippage_cost, 4),
            spread_cost=round(spread_cost, 4),
            commission=round(commission, 4),
            total_cost=round(total_cost, 4),
            liquidity_tier=tier,
            partial_fill=partial,
            notes=notes,
        )

    @staticmethod
    def _liquidity_tier(avg_volume: float) -> str:
        if avg_volume >= LIQUIDITY_HIGH_THRESHOLD:
            return "HIGH"
        elif avg_volume >= LIQUIDITY_MEDIUM_THRESHOLD:
            return "MEDIUM"
        return "LOW"


def _result_to_dict(r: ExecutionResult) -> dict:
    return {
        "symbol":           r.symbol,
        "side":             r.side,
        "requested_shares": r.requested_shares,
        "filled_shares":    r.filled_shares,
        "fill_pct":         round(r.fill_pct, 4),
        "requested_price":  r.requested_price,
        "executed_price":   r.executed_price,
        "slippage_cost":    r.slippage_cost,
        "spread_cost":      r.spread_cost,
        "commission":       r.commission,
        "total_cost":       r.total_cost,
        "liquidity_tier":   r.liquidity_tier,
        "partial_fill":     r.partial_fill,
        "notes":            r.notes,
    }
