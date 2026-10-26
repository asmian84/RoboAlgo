"""
trade_engine/allocator.py
────────────────────────────────────────────────────────────────────────────────
Trade Allocation Engine.

Takes ranked signals from ScoringEngine and converts them into sized trade plans.

Allocation logic:
  • Rank signals by score descending
  • Only top N trades selected (default: 3)
  • Position size proportional to score
  • Hard constraints enforced before any allocation

Constraints:
  • max_risk_per_trade:  2% of account per trade
  • max_open_trades:     3 simultaneous positions
  • min_score_threshold: 0.52 — discard anything below medium confidence
  • max_position_pct:    25% of account in any single position

Position size formula:
  dollar_risk    = account_size × max_risk_pct         e.g. $200 on $10k
  position_size  = account_size × score × risk_factor  scaled by signal quality
  stop_distance  = atr × stop_atr_mult                 price distance to stop
  shares         = position_size / price

Output per allocated trade:
  {
    "symbol":          str,
    "signal_type":     str,      # long | short
    "confidence":      str,      # high | medium | low
    "score":           float,
    "price":           float,
    "shares":          int,
    "position_value":  float,    # shares × price
    "position_pct":    float,    # % of account
    "dollar_risk":     float,    # max loss in $
    "stop_price":      float,
    "target_price":    float,
    "risk_reward":     float,
    "rank":            int,
  }
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Default parameters ────────────────────────────────────────────────────────
DEFAULT_ACCOUNT_SIZE     = 10_000.0
DEFAULT_MAX_RISK_PCT     = 0.02       # 2% risk per trade
DEFAULT_RISK_FACTOR      = 0.10       # base position = account × 10%
DEFAULT_MAX_TRADES       = 3
DEFAULT_MIN_SCORE        = 0.40   # lowered to match scan pipeline (no live GEX in batch)
DEFAULT_MAX_POSITION_PCT = 0.25       # max 25% of account in one position
DEFAULT_STOP_ATR_MULT    = 1.5        # stop = 1.5 × ATR from entry
DEFAULT_TARGET_RR        = 2.0        # target = 2:1 risk-reward


@dataclass
class TradeAllocation:
    symbol:         str
    signal_type:    str       # long | short
    confidence:     str
    score:          float
    price:          float
    shares:         int
    position_value: float
    position_pct:   float     # % of account
    dollar_risk:    float
    stop_price:     float
    target_price:   float
    risk_reward:    float
    rank:           int
    gex_context:    dict = field(default_factory=dict)
    explanation:    list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol":         self.symbol,
            "signal_type":    self.signal_type,
            "confidence":     self.confidence,
            "score":          round(self.score, 4),
            "price":          round(self.price, 4),
            "shares":         self.shares,
            "position_value": round(self.position_value, 2),
            "position_pct":   round(self.position_pct * 100, 2),
            "dollar_risk":    round(self.dollar_risk, 2),
            "stop_price":     round(self.stop_price, 4),
            "target_price":   round(self.target_price, 4),
            "risk_reward":    round(self.risk_reward, 2),
            "rank":           self.rank,
            "gex_context":    self.gex_context,
            "explanation":    self.explanation,
        }


class TradeAllocator:
    """
    Convert scored signals into sized, risk-managed trade plans.

    Usage:
        allocator = TradeAllocator(account_size=25_000)
        trades    = allocator.allocate(scored_signals, market_data)
    """

    def __init__(
        self,
        account_size:     float = DEFAULT_ACCOUNT_SIZE,
        max_risk_pct:     float = DEFAULT_MAX_RISK_PCT,
        risk_factor:      float = DEFAULT_RISK_FACTOR,
        max_trades:       int   = DEFAULT_MAX_TRADES,
        min_score:        float = DEFAULT_MIN_SCORE,
        max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
        stop_atr_mult:    float = DEFAULT_STOP_ATR_MULT,
        target_rr:        float = DEFAULT_TARGET_RR,
    ):
        self.account_size     = account_size
        self.max_risk_pct     = max_risk_pct
        self.risk_factor      = risk_factor
        self.max_trades       = max_trades
        self.min_score        = min_score
        self.max_position_pct = max_position_pct
        self.stop_atr_mult    = stop_atr_mult
        self.target_rr        = target_rr

    def allocate(
        self,
        signals: list[Any],   # list of SignalScore objects or dicts
        market_data: dict | None = None,  # { symbol: { atr, price, ... } }
    ) -> list[TradeAllocation]:
        """
        Allocate capital to top-N signals.

        Args:
            signals:     Sorted list of SignalScore objects (or dicts)
            market_data: Optional { symbol: { atr, price } } for stop calculation

        Returns:
            List of TradeAllocation objects, max length = max_trades
        """
        market_data = market_data or {}
        allocations: list[TradeAllocation] = []
        deployed_capital = 0.0

        for rank, sig in enumerate(signals, start=1):
            if len(allocations) >= self.max_trades:
                break

            # Normalize signal to dict
            if hasattr(sig, "to_dict"):
                s = sig.to_dict()
            elif isinstance(sig, dict):
                s = sig
            else:
                continue

            score = float(s.get("score", 0))
            if score < self.min_score:
                continue

            symbol      = s.get("symbol", "")
            signal_type = s.get("signal_type", "long")
            price       = float(s.get("price", 0))
            confidence  = s.get("confidence", "low")

            if price <= 0:
                # Try to get price from market_data
                price = float(market_data.get(symbol, {}).get("price", 0))
            if price <= 0:
                logger.warning("No price for %s — skipping", symbol)
                continue

            # ── ATR for stop placement ────────────────────────────────────
            atr = float(market_data.get(symbol, {}).get("atr", price * 0.02))
            stop_distance = atr * self.stop_atr_mult

            # ── Position sizing ───────────────────────────────────────────
            # Score-scaled position: higher score → bigger position
            raw_position = self.account_size * self.risk_factor * score

            # Risk constraint: dollar_risk ≤ account × max_risk_pct
            max_dollar_risk = self.account_size * self.max_risk_pct
            # Max shares based on risk budget: risk per share = stop_distance
            max_shares_by_risk = int(max_dollar_risk / stop_distance) if stop_distance > 0 else 0

            # Max shares based on position cap
            max_position_value = self.account_size * self.max_position_pct
            max_shares_by_cap  = int(max_position_value / price) if price > 0 else 0

            # Max shares based on remaining capital
            remaining = self.account_size - deployed_capital
            max_shares_by_capital = int(min(raw_position, remaining) / price) if price > 0 else 0

            shares = max(0, min(
                max_shares_by_risk,
                max_shares_by_cap,
                max_shares_by_capital,
            ))

            if shares == 0:
                logger.info("No shares allocated for %s (constraints hit)", symbol)
                continue

            position_value = shares * price
            position_pct   = position_value / self.account_size
            dollar_risk    = shares * stop_distance

            # ── Stop and target ───────────────────────────────────────────
            if signal_type == "long":
                stop_price   = price - stop_distance
                target_price = price + stop_distance * self.target_rr
            else:  # short
                stop_price   = price + stop_distance
                target_price = price - stop_distance * self.target_rr

            risk_reward = self.target_rr  # by construction

            deployed_capital += position_value

            allocation = TradeAllocation(
                symbol         = symbol,
                signal_type    = signal_type,
                confidence     = confidence,
                score          = score,
                price          = price,
                shares         = shares,
                position_value = position_value,
                position_pct   = position_pct,
                dollar_risk    = dollar_risk,
                stop_price     = stop_price,
                target_price   = target_price,
                risk_reward    = risk_reward,
                rank           = rank,
                gex_context    = s.get("gex_context", {}),
                explanation    = s.get("explanation", []),
            )
            allocations.append(allocation)
            logger.info(
                "Allocated %s: %d shares @ $%.2f | risk=$%.2f | target=$%.2f",
                symbol, shares, price, dollar_risk, target_price,
            )

        return allocations

    def summary(self, allocations: list[TradeAllocation]) -> dict[str, Any]:
        """Return portfolio-level summary of all allocations."""
        if not allocations:
            return {"trades": 0, "deployed": 0, "total_risk": 0, "cash_remaining": self.account_size}
        deployed    = sum(a.position_value for a in allocations)
        total_risk  = sum(a.dollar_risk for a in allocations)
        cash        = self.account_size - deployed
        avg_score   = float(np.mean([a.score for a in allocations]))
        return {
            "trades":          len(allocations),
            "deployed":        round(deployed, 2),
            "deployed_pct":    round(deployed / self.account_size * 100, 2),
            "total_risk":      round(total_risk, 2),
            "total_risk_pct":  round(total_risk / self.account_size * 100, 2),
            "cash_remaining":  round(cash, 2),
            "avg_score":       round(avg_score, 4),
            "symbols":         [a.symbol for a in allocations],
        }


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate scored signals from ScoringEngine
    mock_signals = [
        {"symbol": "NVDA", "score": 0.88, "signal_type": "long",  "confidence": "high",
         "price": 875.0, "explanation": ["Gamma squeeze setup", "Strong momentum"]},
        {"symbol": "SPY",  "score": 0.74, "signal_type": "long",  "confidence": "medium",
         "price": 502.0, "explanation": ["Near gamma flip"]},
        {"symbol": "QQQ",  "score": 0.71, "signal_type": "short", "confidence": "medium",
         "price": 435.0, "explanation": ["Breakdown below put wall"]},
        {"symbol": "TSLA", "score": 0.45, "signal_type": "long",  "confidence": "low",
         "price": 178.0, "explanation": ["Weak setup"]},
    ]

    market_data = {
        "NVDA": {"atr": 18.5, "price": 875.0},
        "SPY":  {"atr": 4.2,  "price": 502.0},
        "QQQ":  {"atr": 5.1,  "price": 435.0},
        "TSLA": {"atr": 9.2,  "price": 178.0},
    }

    allocator = TradeAllocator(account_size=25_000)
    trades    = allocator.allocate(mock_signals, market_data)

    print(f"\n{'='*60}")
    print(f"TRADE ALLOCATION REPORT — Account: ${allocator.account_size:,.0f}")
    print(f"{'='*60}")
    for t in trades:
        d = t.to_dict()
        print(f"\n#{d['rank']} {d['symbol']} ({d['signal_type'].upper()}) — Score: {d['score']:.2f} [{d['confidence']}]")
        print(f"   Entry: ${d['price']}  Shares: {d['shares']}  Value: ${d['position_value']:,.0f} ({d['position_pct']:.1f}%)")
        print(f"   Stop:  ${d['stop_price']:.2f}  Target: ${d['target_price']:.2f}  RR: {d['risk_reward']:.1f}:1")
        print(f"   Risk:  ${d['dollar_risk']:.2f}")

    print(f"\n{'─'*60}")
    s = allocator.summary(trades)
    print(f"Deployed: ${s['deployed']:,.0f} ({s['deployed_pct']:.1f}%)  "
          f"Total Risk: ${s['total_risk']:,.0f} ({s['total_risk_pct']:.1f}%)  "
          f"Cash: ${s['cash_remaining']:,.0f}")
