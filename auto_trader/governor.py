"""
governor — pure risk policy for auto_trader.

ZERO I/O: no DB, no network, no Redis, no file reads. This module only takes
plain-old-data inputs and returns a verdict. It is the leaf module other
auto_trader modules import from.

Rules are evaluated in strict precedence order (first match wins):
    1. kill_switch
    2. daily_loss_limit
    3. consecutive_loss_halt
    4. invalid_order
    5. duplicate_position
    6. max_open_positions
    7. max_position_size
    8. per_trade_risk
    9. max_total_exposure
    10. insufficient_buying_power
    else -> approved
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RiskLimits:
    max_open_positions: int = 5
    max_total_exposure_pct: float = 60.0
    per_trade_risk_pct: float = 1.0
    daily_loss_limit_pct: float = 3.0
    consecutive_loss_halt: int = 3
    max_position_size_pct: float = 20.0


@dataclass
class PositionView:
    symbol: str
    qty: int
    market_value: float
    unrealized_pnl: float
    side: str = "long"


@dataclass
class PortfolioSnapshot:
    equity: float
    cash: float
    positions: list[PositionView]
    day_start_equity: float
    consecutive_losses: int

    @property
    def open_symbols(self) -> set[str]:
        return {p.symbol.upper() for p in self.positions}

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions)


@dataclass
class CandidateOrder:
    symbol: str
    side: str
    qty: int
    entry: float
    stop: float
    target: float
    notional: float
    risk_amount: float
    score: float = 0.0
    confidence: float = 0.0


@dataclass
class GovernorVerdict:
    approved: bool
    rule: Optional[str]
    reason: str


def check_halts(
    snapshot: PortfolioSnapshot, limits: RiskLimits, kill_switch_on: bool
) -> GovernorVerdict:
    """Rules 1-3: account-wide halts. No knowledge of any specific order."""

    if kill_switch_on:
        return GovernorVerdict(False, "kill_switch", "kill switch is engaged")

    if snapshot.day_start_equity > 0:
        day_pnl_pct = (
            (snapshot.equity - snapshot.day_start_equity)
            / snapshot.day_start_equity
            * 100
        )
        if day_pnl_pct <= -limits.daily_loss_limit_pct:
            return GovernorVerdict(
                False,
                "daily_loss_limit",
                f"day pnl {day_pnl_pct:.2f}% <= -{limits.daily_loss_limit_pct:.2f}% "
                f"(equity {snapshot.equity:.2f}, day_start_equity "
                f"{snapshot.day_start_equity:.2f})",
            )

    if snapshot.consecutive_losses >= limits.consecutive_loss_halt:
        return GovernorVerdict(
            False,
            "consecutive_loss_halt",
            f"consecutive_losses {snapshot.consecutive_losses} >= "
            f"consecutive_loss_halt {limits.consecutive_loss_halt}",
        )

    return GovernorVerdict(True, None, "no halts active")


def check_order(
    candidate: CandidateOrder, snapshot: PortfolioSnapshot, limits: RiskLimits
) -> GovernorVerdict:
    """Rules 4-10: per-order checks against the current portfolio state."""

    if (
        candidate.qty <= 0
        or candidate.entry <= 0
        or candidate.stop <= 0
        or (candidate.side == "buy" and candidate.stop >= candidate.entry)
        or (candidate.side == "sell" and candidate.stop <= candidate.entry)
    ):
        return GovernorVerdict(
            False,
            "invalid_order",
            f"invalid order: side={candidate.side} qty={candidate.qty} "
            f"entry={candidate.entry:.2f} stop={candidate.stop:.2f}",
        )

    if candidate.symbol.upper() in snapshot.open_symbols:
        return GovernorVerdict(
            False,
            "duplicate_position",
            f"symbol {candidate.symbol.upper()} already has an open position",
        )

    if len(snapshot.positions) >= limits.max_open_positions:
        return GovernorVerdict(
            False,
            "max_open_positions",
            f"open positions {len(snapshot.positions)} >= "
            f"max_open_positions {limits.max_open_positions}",
        )

    max_position_size = snapshot.equity * limits.max_position_size_pct / 100
    if candidate.notional > max_position_size:
        return GovernorVerdict(
            False,
            "max_position_size",
            f"notional {candidate.notional:.2f} > max_position_size "
            f"{max_position_size:.2f} ({limits.max_position_size_pct}% of "
            f"{snapshot.equity:.2f})",
        )

    max_risk = snapshot.equity * limits.per_trade_risk_pct / 100
    if candidate.risk_amount > max_risk:
        return GovernorVerdict(
            False,
            "per_trade_risk",
            f"risk_amount {candidate.risk_amount:.2f} > per_trade_risk "
            f"{max_risk:.2f} ({limits.per_trade_risk_pct}% of "
            f"{snapshot.equity:.2f})",
        )

    max_exposure = snapshot.equity * limits.max_total_exposure_pct / 100
    projected_exposure = snapshot.gross_exposure + candidate.notional
    if projected_exposure > max_exposure:
        return GovernorVerdict(
            False,
            "max_total_exposure",
            f"gross_exposure {snapshot.gross_exposure:.2f} + notional "
            f"{candidate.notional:.2f} = {projected_exposure:.2f} > "
            f"max_total_exposure {max_exposure:.2f} "
            f"({limits.max_total_exposure_pct}% of {snapshot.equity:.2f})",
        )

    if candidate.notional > snapshot.cash:
        return GovernorVerdict(
            False,
            "insufficient_buying_power",
            f"notional {candidate.notional:.2f} > cash {snapshot.cash:.2f}",
        )

    return GovernorVerdict(True, None, "approved")


def evaluate(
    candidate: CandidateOrder,
    snapshot: PortfolioSnapshot,
    limits: RiskLimits,
    kill_switch_on: bool,
) -> GovernorVerdict:
    """Full pipeline: account halts first, then per-order checks."""

    halt_verdict = check_halts(snapshot, limits, kill_switch_on)
    if not halt_verdict.approved:
        return halt_verdict

    return check_order(candidate, snapshot, limits)
